// Linux driver for FPGA accelerator for convolutional neural network used for recognition of motor vehicles

/* Headers */

#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/of_address.h>
#include <linux/of_device.h>
#include <linux/of_platform.h>
#include <linux/version.h>
#include <linux/types.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/delay.h>

MODULE_AUTHOR("y23-g02 - Ivan David Ivan");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("CNN IP core driver");

#define DRIVER_NAME "cnn"

static dev_t my_dev_id;
static struct class *my_class;
static struct cdev  *my_cdev;

static int cnn_probe     (struct platform_device *pdev);
static int cnn_remove    (struct platform_device *pdev);
static int cnn_open      (struct inode *pinode, struct file *pfile);
static int cnn_close     (struct inode *pinode, struct file *pfile);
static ssize_t cnn_read  (struct file *pfile, char __user *buffer, size_t length, loff_t *offset);
static ssize_t cnn_write (struct file *pfile, const char __user *buffer, size_t length, loff_t *offset);

static int  __init cnn_init(void);
static void __exit cnn_exit(void);

struct device_info
{
    unsigned long mem_start;
    unsigned long mem_end;
    void __iomem *base_addr;
};

static struct device_info *cnn = NULL;
static struct device_info *dma = NULL;

static struct of_device_id device_of_match[] = {
    { .compatible = "xlnx,cnn", },
    { .compatible = "xlnx,dma", },
    { /* end of list */ }
};

MODULE_DEVICE_TABLE(of, device_of_match);

static struct platform_driver my_driver = {
    .driver = {
        .name = DRIVER_NAME,
        .owner = THIS_MODULE,
        .of_match_table	= device_of_match,
    },
    .probe = cnn_probe,
    .remove	= cnn_remove,
};

struct file_operations my_fops =
{
	.owner   = THIS_MODULE,
	.open    = cnn_open,
	.read    = cnn_read,
	.write   = cnn_write,
	.release = cnn_close,
};


/* -------------------------------------- */

/* Init function being called and executed only once by insmod command. */

static int __init cnn_init(void)
{
    printk(KERN_INFO "\nInitializing CNN...\n");


    /* Dynamically allocate MAJOR and MINOR numbers. */

    if(alloc_chrdev_region(&my_dev_id, 0, 2, "cnn_region") < 0)
    {
        printk(KERN_ERR "Failed to register char device.\n");
        return -1;
    }
    printk(KERN_INFO "Char device region allocated.\n");

    /* Creating NODE files */

    /* First, class_create is used to create class to be used as a parametar going forward. */

    my_class = class_create(THIS_MODULE, "cnn_class");
    if(my_class == NULL)
    {
        printk(KERN_ERR "Failed to create class.\n");
        goto error_0;
    }
    printk(KERN_INFO "Class created.\n");

    /* Secondly, device_create is used to create devices in a region. 
    Results in two files that represent hardware components that cna be seen as a programmers view. */


    /* Creating IP. */

    if(device_create(my_class, NULL, MKDEV(MAJOR(my_dev_id), 0), NULL, "xlnx,cnn") == NULL) 
    {
        printk(KERN_ERR "Failed to create device cnn.\n");
        goto error_1;
    }
    printk(KERN_INFO "Device created - cnn.\n");

    /* Creating DMA. */

    if(device_create(my_class, NULL, MKDEV(MAJOR(my_dev_id), 1), NULL, "xlnx,dma") == NULL) 
    {
        printk(KERN_ERR "Failed to create device dma.\n");
        goto error_2;
    }
    printk(KERN_INFO "Device created - dma.\n");


	my_cdev = cdev_alloc();
	my_cdev->ops = &my_fops;
	my_cdev->owner = THIS_MODULE;

    if(cdev_add(my_cdev, my_dev_id, 2) == -1)
    {
        printk(KERN_ERR "Failed to add cdev.\n");
        goto error_3;
    }
    printk(KERN_INFO "cdev added\n");
    printk(KERN_INFO "CNN driver initialized.\n");

    return platform_driver_register(&my_driver);

    /* If anything goes wrong, undone all that has been done so far. */

    error_3:
        device_destroy(my_class, MKDEV(MAJOR(my_dev_id), 1));
    error_2:
        device_destroy(my_class, MKDEV(MAJOR(my_dev_id), 0));
    error_1:
        class_destroy(my_class);
    error_0:
        unregister_chrdev_region(my_dev_id, 1);
    return -1;

}

/* Exit function being called and executed only once by rmmod command. */

static void __exit cnn_exit(void)
{
    printk(KERN_INFO "CNN driver starting rmmod...\n");
    platform_driver_unregister(&my_driver);
    cdev_del(my_cdev);
    device_destroy(my_class, MKDEV(MAJOR(my_dev_id), 1));
    device_destroy(my_class, MKDEV(MAJOR(my_dev_id), 0));
    class_destroy(my_class);
    unregister_chrdev_region(my_dev_id, 1);
    printk(KERN_INFO "CNN driver exited.\n");
}

module_init(cnn_init);
module_exit(cnn_exit);  